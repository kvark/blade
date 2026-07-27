# Collecting synchronization-study data

## Frozen sources

The working benchmark branches are:

- Blade: `blade-sync-study`, based on
  `f724f6a9da89939de908ec434e65e86a9a789477`.
- wgpu: `blade-sync-study`, based on the `v30` release branch at
  `1464540aaf62667837b7a5f0934f2e33b9a39b43`.

Final results must use commits made from the current working trees, not these
base commits. Record the resulting two commit IDs in the paper and pin them in
the study repository.

Until the study repository exists, check out the repositories as siblings:

```text
Code/
  blade/
  wgpu/
```

All commands below start in the Blade checkout. The collector records both
revisions, dirty status, lock-file hashes, hardware metadata, randomized order,
commands, selected adapters, validation hashes, and raw rows.

## Machines to cover

For the core Vulkan paper, the minimum defensible set is:

1. one recent discrete NVIDIA GPU;
2. one recent discrete AMD GPU;
3. the same operating-system family on both machines, preferably Linux if both
   Vulkan driver stacks are available there.

This supports statements about the measured device-driver pairs, not NVIDIA or
AMD in general. The preferred set adds one older architecture from each vendor
and repeats one AMD and one NVIDIA GPU on Windows. Windows is not required
unless the paper makes a cross-OS or cross-driver-stack claim.

Run the matched Metal matrix on one Apple Silicon machine as a separate case
study. Do not pool it with Vulkan results. Intel, integrated/UMA Vulkan, and
additional Apple generations are useful sensitivity cases, not blockers for
the initial report.

Thus the practical collection order is:

1. NVIDIA/Linux and AMD/Linux;
2. Apple Silicon/macOS;
3. older AMD/NVIDIA generations;
4. Windows repeats.

## Select adapters

Build and list the Blade adapters:

```sh
cargo run --release --example sync-bench -- --list-adapters
```

List wgpu Vulkan adapters on Linux:

```sh
WGPU_BACKEND=vulkan \
cargo run --release -p wgpu-sync-bench \
  --manifest-path ../wgpu/Cargo.toml -- --list-adapters
```

List wgpu Metal adapters on macOS:

```sh
WGPU_BACKEND=metal \
cargo run --release -p wgpu-sync-bench \
  --manifest-path ../wgpu/Cargo.toml -- --list-adapters
```

Use Blade's hexadecimal device ID with `--blade-device-id`. Select the same
physical adapter in wgpu with a unique case-insensitive name substring passed
to `--wgpu-adapter-name`.

On Windows PowerShell:

```powershell
cargo run --release --example sync-bench -- --list-adapters
$env:WGPU_BACKEND = "vulkan"
cargo run --release -p wgpu-sync-bench `
  --manifest-path ..\wgpu\Cargo.toml -- --list-adapters
Remove-Item Env:WGPU_BACKEND
```

## Correctness collection

Run this once after every driver, OS, or benchmark change. Replace the adapter
values and output label.

Linux Vulkan:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend vulkan \
  --blade-device-id 0x2f04 \
  --wgpu-adapter-name "NVIDIA GeForce RTX 5070" \
  --output paper/data/raw/linux-nvidia-correctness \
  --repetitions 1 \
  --passes 4 \
  --elements 65536 \
  --rounds 2 \
  --width 256 \
  --height 256 \
  --warmups 0 \
  --samples 3 \
  --validation \
  --cpu-only
```

Windows PowerShell:

```powershell
py -3 paper\run-study-matrix.py `
  --wgpu ..\wgpu `
  --backend vulkan `
  --blade-device-id 0x2f04 `
  --wgpu-adapter-name "RTX 5070" `
  --output paper\data\raw\windows-nvidia-correctness `
  --repetitions 1 `
  --passes 4 `
  --elements 65536 `
  --rounds 2 `
  --width 256 `
  --height 256 `
  --warmups 0 `
  --samples 3 `
  --validation `
  --cpu-only
```

Replace only the two adapter selectors and the host label. The collector
itself does not require Bash.

macOS Metal:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend metal \
  --wgpu-adapter-name "Apple M3" \
  --output paper/data/raw/macos-apple-m3-correctness \
  --repetitions 1 \
  --passes 4 \
  --elements 65536 \
  --rounds 2 \
  --width 256 \
  --height 256 \
  --warmups 0 \
  --samples 3 \
  --validation \
  --cpu-only
```

The collector fails if Blade policies or wgpu produce different output hashes
for the same workload.

### Vulkan synchronization validation

Matching output hashes are necessary but not sufficient for a policy that
narrows synchronization scopes: a missing dependency can go unobserved on one
schedule. On Vulkan, `run-study-matrix.py --validation` forces the Khronos
validation layer and enables its synchronization checks for every Blade policy
and the matched wgpu workload. If the layer is unavailable, the process fails
instead of silently producing an unvalidated collection. Install the layer,
then run the correctness command above:

```sh
sudo apt install vulkan-validationlayers   # or the platform equivalent
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend vulkan \
  --blade-device-id 0x2f04 \
  --wgpu-adapter-name "RTX 5070" \
  --repetitions 1 \
  --passes 4 \
  --elements 65536 \
  --rounds 2 \
  --width 256 \
  --height 256 \
  --warmups 0 \
  --samples 3 \
  --validation \
  --cpu-only
```

The runner retains each process's combined validation output as
`*.validation.txt`, records that file and the forced environment in the
manifest, and keeps validation callback text out of the parseable benchmark
CSV. It scans for `SYNC-HAZARD` and `Validation Error` and aborts if either
appears. A `SYNC-HAZARD-*` message means the derived stage or access masks are
too narrow for that workload; any other validation error also disqualifies the
executable from performance collection until it is understood and fixed.
This check has found two real defects: Blade's `extra_sync_*_access` driver
workarounds named transfer accesses that only `ALL_COMMANDS` supports, and the
retained wgpu graphics shader produced invalid SPIR-V from a function-local
array. The current array-free shader passes the full small matrix.

`paper/collect.py` runs this correctness matrix automatically before the
performance matrix. Pass `--skip-validation` only when a retained validation
collection already covers the exact sources, driver, and OS.

## Final CPU collection

GPU queries are disabled in this collection so their unequal API plumbing does
not affect command-recording measurements:

Linux NVIDIA example:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend vulkan \
  --blade-device-id 0x2f04 \
  --wgpu-adapter-name "RTX 5070" \
  --output paper/data/raw/linux-nvidia-cpu \
  --repetitions 10 \
  --cpu-only
```

Use the same command on the Linux AMD machine, replacing the device ID, unique
adapter-name substring, and output directory with
`paper/data/raw/linux-amd-cpu`.

Windows PowerShell:

```powershell
py -3 paper\run-study-matrix.py `
  --wgpu ..\wgpu `
  --backend vulkan `
  --blade-device-id 0x2f04 `
  --wgpu-adapter-name "RTX 5070" `
  --output paper\data\raw\windows-nvidia-cpu `
  --repetitions 10 `
  --cpu-only
```

macOS Apple Silicon:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend metal \
  --wgpu-adapter-name "Apple M3" \
  --output paper/data/raw/macos-apple-m3-cpu \
  --repetitions 10 \
  --cpu-only
```

Replace `Apple M3` and the output label with the adapter reported by the
discovery command. Add `--blade-device-id` only if Blade enumerates more than
one selectable Apple GPU.

## Final GPU collection

This collection enables timestamp queries. Host times from this run are
secondary because Blade and wgpu retrieve timestamps differently.

Linux NVIDIA example:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend vulkan \
  --blade-device-id 0x2f04 \
  --wgpu-adapter-name "RTX 5070" \
  --output paper/data/raw/linux-nvidia-gpu \
  --repetitions 10
```

Use the corresponding AMD selectors and `linux-amd-gpu` on the Linux AMD
machine.

Windows PowerShell:

```powershell
py -3 paper\run-study-matrix.py `
  --wgpu ..\wgpu `
  --backend vulkan `
  --blade-device-id 0x2f04 `
  --wgpu-adapter-name "RTX 5070" `
  --output paper\data\raw\windows-nvidia-gpu `
  --repetitions 10
```

macOS Apple Silicon:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend metal \
  --wgpu-adapter-name "Apple M3" \
  --output paper/data/raw/macos-apple-m3-gpu \
  --repetitions 10
```

The wgpu benchmark uses beginning/end timestamps on the first and last WebGPU
passes. This requires `TIMESTAMP_QUERY`, which wgpu exposes on supported Apple
GPUs; it does not require the encoder/inside-pass timestamp features that Apple
GPUs lack. If either implementation reports that timestamp queries are
unavailable, retain the CPU collection and use an external GPU profiler rather
than substituting one implementation's timing method.

Do not pass `--allow-dirty` for final data. It exists only for local pilots.
The collector refuses an existing output directory and retains stderr from any
run that emits it.

## Let the device reach a steady clock first

One block is one process launch, so a device may idle between configurations
and ramp again inside the next one. The pilot's ten warm-ups left the RX 7900
XT accelerating throughout its 30 measured iterations. The current fixed
matrix uses 1000 warm-ups. Do not infer that a universal count exists: inspect
time-ordered samples and compare each block's first and last thirds.

**Warm up until drift stops.** Start with the current setting:

```sh
python3 paper/collect.py --wgpu ../wgpu --warmups 1000 --repetitions 10
```

Increase the warm-up if any block still moves. `build-tables.py` reports
first-third versus last-third drift and the per-cell control intervals, so
read those generated diagnostics rather than a copied table of old values.

**Pinning clocks helps and does not suffice.** The RX 7900 XT collection ran
with `power_dpm_force_performance_level = high` on both of its devices --
`rocm-smi` recorded `Performance Level: high` -- and still drifted in the
ten-warm-up pilot. The same capture shows a 35 MHz shader clock at idle
underneath the setting: `high` raises the ceiling, it does not hold the floor.
Set it anyway, and do not treat it as sufficient.

On AMD, before collecting:

```sh
for card in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  echo high | sudo tee "$card"
done
```

On NVIDIA:

```sh
sudo nvidia-smi --lock-gpu-clocks=tdp --mode=1
```

On Intel, `intel_gpu_frequency -s <MHz>` from `intel-gpu-tools` sets a fixed
frequency; equivalently write the same value to `gt_min_freq_mhz` and
`gt_max_freq_mhz` under `/sys/class/drm/card*/`, or to `freq0/min_freq` and
`freq0/max_freq` under the `xe` driver's tile tree. Restore with `auto` /
`--reset-gpu-clocks` afterwards.

On a power-limited mobile part, lock to a frequency the device sustains rather
than to its peak: pinning to `RP0` invites the package limit to throttle
mid-collection, which puts a step back into the block.

The CPU governor matters for host cost and not for device span. Where host
numbers are wanted, use `sudo cpupower frequency-set -g performance` and a
quiet machine, and retain the setting in the collection.

`run-study-matrix.py` reads these knobs itself, writes them to
`power-state.txt`, records them in the manifest as `power_state` with a
`clocks_locked` summary, and warns before collecting if a GPU is still under
automatic management. Nothing has to be remembered or written down by hand.

Then check the stability diagnostic before interpreting an effect. A
directional effect must have a hierarchical interval excluding zero and exceed
the conservative post-hoc threshold derived from its paired global
explicit-all control. That threshold is a screening rule, not a calibrated
equivalence or detection bound.

## Barrier policies and workloads

On Vulkan the collector crosses three placements with two scopes, giving six
Blade policies per workload:

| Policy | Placement | Scope |
|---|---|---|
| `automatic` | encoder, every pass | `Global` |
| `automatic-scoped` | encoder, every pass | `PassKind` |
| `hazard-only` | application, dependent workloads only | `Global` |
| `hazard-only-scoped` | application, dependent workloads only | `PassKind` |
| `explicit-all` | application, every pass | `Global` |
| `explicit-all-scoped` | application, every pass | `PassKind` |

Reading down a scope column isolates placement; reading across a placement row
isolates the scope change appropriate to that placement. Only global
`explicit-all` against global `automatic` is the identical-barrier
instrumentation control. At pass-kind scope an explicit barrier must retain an
`ALL_COMMANDS` destination because its consumer is unknown, whereas an
automatic barrier derives both sides. Metal collects `automatic` only, since
`barrier_scope` and `manual_barriers` are Vulkan concepts.

Workloads are the four shared ones plus `mixed-independent` and `mixed-chain`,
which alternate compute and render passes. The mixed pair is Blade-only, so
wgpu still runs four workloads per repetition and Blade runs six. In
`mixed-chain`, each pass consumes the output of the same pass kind two
positions earlier. The benchmark's hazard-only policy nevertheless inserts
before every pass after the first, so it is a conservative application policy,
not a minimal placement oracle, for that one workload.

## Analyze

Analyze each CPU and GPU collection separately:

```sh
python3 paper/analyze.py paper/data/raw/linux-nvidia-cpu \
  --output paper/data/derived/linux-nvidia-cpu

python3 paper/analyze.py paper/data/raw/linux-nvidia-gpu \
  --output paper/data/derived/linux-nvidia-gpu
```

`comparisons.csv` contains the within-Blade policy comparisons and the matched
Blade-automatic versus wgpu-tracked comparison.

Inspect time-ordered raw samples before interpreting aggregates. Repeat a
representative collection after a reboot or on another day.

## CPU profiles

`paper/profile-hosts.py` attributes whole-process CPU samples to the crate or
library in which they landed. It is a diagnostic; in its current form it does
not decompose the measured record-and-submit interval.

```sh
sudo sysctl kernel.perf_event_paranoid=1     # until reboot
python3 paper/profile-hosts.py --wgpu ../wgpu
```

It refuses to run rather than emit an empty profile if `perf` is missing or
`perf_event_paranoid` is above 2. The workload uses 4096 elements, 64x64
targets, one mixing round, 64 passes, and no timestamp queries, but each
iteration still waits for completion. Existing profiles are consequently
dominated by driver/kernel wait activity. A publication-quality host
attribution run must gate `perf` to record plus submit or batch many recordings
before one wait. The script rebuilds both binaries with
`CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`, which improves symbol
attribution without changing optimisation.

Output is `symbols.csv` (self time per symbol), `buckets.csv` (self time per
group), the raw `perf report` text, and a manifest. The groups separate
`wgpu_core::track` from `wgpu_core::init_tracker`: the first is resource state
tracking, the second is lazy zero-initialisation, and conflating them would
overstate the number the profile exists to measure.

Further caveats to record with the result. Self time is attributed to the symbol
the sample landed in, so aggressive inlining moves tracker work into its
callers and can make the tracker-labelled share an undercount. It is not a
formal lower bound: classification and the wait-dominated denominator can move
the share in either direction. The retained profiles were not adapter-pinned
and selected different GPUs for Blade and wgpu on the dual-GPU AMD host; pass
both selectors to the current script, which now refuses to write a manifest or
derived tables when the reported device names differ. A flat profile answers
"where is time spent", not "why" -- a call-graph run (`--call-graph dwarf`)
costs more and needs frame pointers to be reliable.

## GPU captures

`paper/capture-streams.py` captures each Vulkan configuration and extracts its
barrier records, so the barriers the paper describes from source can be checked
against the capture.

```sh
python3 paper/capture-streams.py --wgpu ../wgpu
```

On a multi-GPU machine, pin both selectors:

```sh
python3 paper/capture-streams.py --wgpu ../wgpu \
  --blade-device-id 0x744c \
  --wgpu-adapter-name 'AMD Radeon RX 7900 XT (RADV NAVI31)'
```

The runner requires exactly one new `.rdc` per requested configuration, records
the benchmark metadata and relevant adapter environment in the manifest,
rejects differing output hashes, and aborts if Blade and wgpu report different
device names. These checks matter even though the barrier requests are expected
to be device-independent: without them, a multi-GPU host is not a reproducible
matched capture.

If no RenderDoc is installed it downloads the official build, verifies it
against a pinned hash, and unpacks it under `paper/data/tools/`. Use
`--no-download` to refuse that, or `--library` to point at your own. Note that
the library ends up preloaded into the benchmark process, which is a trust
decision the script states out loud rather than making quietly.

The benchmark is headless, so there is no swapchain present for RenderDoc to
delimit a capture at; `sync-bench --capture` calls the in-application API
around one warmed iteration instead, and the script preloads
`librenderdoc.so` so that API is available. Without the preload the benchmark
warns and runs uncaptured rather than failing a measurement.

After capturing it converts each `.rdc` with `renderdoccmd convert -c xml` and
tabulates every `vkCmdPipelineBarrier` into `barriers.csv`: stage and access
masks decoded to names, barrier counts by kind, any image layout transition,
and `after_work`, the number of preceding draw/dispatch commands. The last
field makes placement differences visible; equal barrier counts alone do not
establish equal placement. That is what makes the captures checkable rather
than merely archived.

Both implementations are captured: the matched wgpu benchmark takes the same
`--capture` flag and wraps the same warmed iteration, so the extracted tables
come from the same workload on the same machine state. `barriers.csv` is not a
serialization of the complete command stream. wgpu is skipped for the mixed
workloads, which exist only in Blade's benchmark.

A note for anyone extending this: RenderDoc hooks Vulkan through a layer, not
through the preloaded library alone, and the manifest shipped in the tarball
carries the absolute `library_path` of upstream's build machine. Without
rewriting that path and pointing the loader at the manifest, the benchmark runs
happily and writes no capture at all.

## Manual captures and profilers

Build the wgpu benchmark with Tracy scopes enabled:

```sh
cargo build --release -p wgpu-sync-bench \
  --manifest-path ../wgpu/Cargo.toml --features tracy
```

For a Linux `perf` profile that includes wgpu tracker call stacks:

```sh
perf record --call-graph dwarf \
  ../wgpu/target/release/wgpu-sync-bench \
  --workload compute-independent \
  --policy tracked \
  --passes 100 \
  --elements 1048576 \
  --rounds 8 \
  --warmups 10 \
  --samples 100 \
  --no-gpu-timing
```

Set `WGPU_BACKEND` and `WGPU_ADAPTER_NAME` in the environment before launching
the profiler. Capture one independent and one dependent compute workload and
their graphics equivalents in RenderDoc, Radeon GPU Profiler, or Nsight
Graphics. Retain the capture-tool version and capture file alongside the raw
collection.

## Apple hazard-mode experiment

The harness lives with the paper, at `paper/tools/metal-hazard-bench.swift`. It
is standalone Swift with no Blade dependency, so it can be built and run
directly:

```sh
xcrun swiftc -O paper/tools/metal-hazard-bench.swift \
  -o /tmp/metal-hazard-bench
/tmp/metal-hazard-bench \
  --raw-output paper/data/raw/<collection-id>/metal-hazard-r01-raw.csv \
  | tee paper/data/raw/<collection-id>/metal-hazard-r01-summary.csv
```

The required `--raw-output` path must not exist; the harness refuses to
overwrite it. That file contains every observation in execution order. Standard
output contains one summary row per (workload, tracking mode, pass count) with
median and 5th/95th-percentile encode, commit, GPU, and wall times. Both outputs
record the effective `hazardTrackingMode`, device, OS, UTC time, and session ID.
Run at least ten fresh processes (`r01` through `r10`) while on AC power with
Low Power Mode disabled.

The results currently cited by the paper are in
[`../docs/metal-hazard-tracking.md`](../docs/metal-hazard-tracking.md) and were
collected on battery with Low Power Mode enabled and retained only as
transcribed summaries. Repeat as above before using the pilot for inferential or
equivalence claims, and retain the macOS build, hardware report, and Blade
revision alongside the raw files.

## Pass-count sweeps

`--pass-list` replaces `--passes` with a comma-separated sweep and appends
`__pNNNN` to each run ID. Use it for the scaling figures:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend vulkan \
  --blade-device-id 0x2f04 \
  --wgpu-adapter-name "RTX 5070" \
  --output paper/data/raw/<collection-id>-sweep-cpu \
  --pass-list 1,2,4,8,16,32,64,128 \
  --cpu-only
```

Run the same command without `--cpu-only` into a `-sweep` directory for device
time. Host and device claims must come from the respective collection: with
timestamps enabled, Blade writes one query per pass and wgpu writes two in
total, and at high pass counts that alone changes the apparent host cost by a
factor of four.

## Device selection

By default the collector enumerates every adapter both implementations can see
and runs the whole matrix once per device, into
`paper/data/raw/<collection-id>-<hostname>-<device-slug>/`. With a single
device the suffix is dropped and the directory is
`<collection-id>-<hostname>/`. Software adapters are skipped unless
`--allow-software` is given, and an adapter Blade sees but wgpu does not is
skipped with a message.

So on a machine with an integrated and a discrete GPU the whole study is:

```sh
python3 paper/run-study-matrix.py --wgpu ../wgpu --repetitions 10
```

Passing `--blade-device-id` or `--wgpu-adapter-name` pins one device and
restores the previous single-collection behaviour. Either way the collector
aborts if Blade and wgpu end up reporting different `device_name` values, which
is the failure the `rubik` collection hit before this check existed.
