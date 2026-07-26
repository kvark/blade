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

Raw measurements belong under `data/raw/` and generated tables under
`data/derived/`. Neither is tracked by git -- see [data/README.md](data/README.md).
Every result used by the paper must retain the raw rows and machine metadata
from which it was derived, which the collection directory does.

With Blade and the benchmark wgpu branch checked out as sibling directories,
one command collects everything this machine can contribute:

```bash
python3 paper/collect.py --wgpu ../wgpu
```

That runs the timing matrix over every adapter, then a host CPU profile, then
RenderDoc captures of both implementations. The last two need `perf` and
RenderDoc respectively; if either is unavailable it says so and carries on,
because a machine that can only contribute timings should still contribute
them. Add `--sweeps` for the pass-count sweeps, which take roughly another half
hour. Unrecognized arguments pass through to the matrix runner.

The three steps can also be run on their own:

```bash
python3 paper/run-study-matrix.py --wgpu ../wgpu     # timings
python3 paper/profile-hosts.py    --wgpu ../wgpu     # host CPU profile
python3 paper/capture-streams.py  --wgpu ../wgpu     # command streams
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
| `20260725T191617Z-zork` | zork | NVIDIA RTX 5070 | matrix, placement × scope |
| `*-rubik-amd-radeon-rx-7900-xt-radv-n` | rubik | AMD RX 7900 XT | matrix, placement × scope |
| `*-rubik-amd-ryzen-5-9600x-6-core-pro` | rubik | AMD Raphael iGPU | matrix, placement × scope |
| `20260725T204342Z-k6` | k6 | AMD Radeon 780M | matrix, placement × scope |
| `20260725T204044Z-matrix` | matrix | Intel Xe (RPL-U) | matrix, placement × scope (chain cells clean; independent-target cells noisy) |
| `20260725T062529Z-mac` | mac | Apple M3 | matrix, Metal case study |
| `20260725T060107Z-zork` | zork | NVIDIA RTX 5070 | superseded by `191617` |
| `20260725T185508Z-zork` | zork | NVIDIA RTX 5070 | pass-count sweep 1-64, GPU-timed |
| `20260725T190100Z-zork` | zork | NVIDIA RTX 5070 | pass-count sweep 1-64, timestamp-free |

`build-tables.py` keeps only the most recent matrix collection per machine and
device, so a retest supersedes an earlier run rather than appearing beside it.
The two `rubik` directories come from one invocation: the collector runs every
enumerated adapter in turn.

Because `data/raw/` is not in git, a collection is only where it was copied.
`build-tables.py` warns when `main.tex` cites a device whose data is not on the
current disk, since otherwise its table rows vanish while the prose keeps
referring to them. If that warning appears, copy the named collections back
before building the PDF.

The result gates of [experiments.md](experiments.md) are met. CPU profiles
(`profile-hosts.py`) and command-stream captures of both implementations
(`capture-streams.py`) are collected and reported.

The workload set is closed; see [experiments.md](experiments.md) for what was
considered and declined, and why. What remains is filling the existing grid:

| machine | matrix | profile | captures |
|---|:--:|:--:|:--:|
| zork | yes | yes | yes |
| rubik | yes | yes | yes |
| k6 | yes | — | — |
| matrix | yes | yes | yes |
| mac | yes | n/a | n/a |

`python3 paper/collect.py --wgpu ../wgpu` fills a row. Apple has neither a
`perf` nor a RenderDoc path, so its row is complete as it stands.

Optional: clock-locked repeats to recover the cells whose control floor exceeds
the effect in them. Not a blocker — the conclusions rest on cells whose floor is
below 1.5%, and the `ctrl` column of the scope table says which is which.

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
