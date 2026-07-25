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
  --wgpu-adapter-name "RTX 5070" \
  --output paper/data/raw/linux-nvidia-correctness \
  --repetitions 1 \
  --passes 4 \
  --elements 65536 \
  --rounds 2 \
  --width 256 \
  --height 256 \
  --warmups 0 \
  --samples 1 \
  --validation
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
  --samples 1 `
  --validation
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
  --samples 1 \
  --validation
```

The collector fails if Blade policies or wgpu produce different output hashes
for the same workload.

### Synchronization validation for `automatic-scoped`

Matching output hashes are necessary but not sufficient for a policy that
narrows synchronization scopes: a missing dependency can go unobserved on one
schedule. Before trusting `automatic-scoped` numbers from a machine, run the
correctness collection there with the Khronos validation layer's
synchronization checks enabled:

```sh
sudo apt install vulkan-validationlayers   # or the platform equivalent
for w in compute-independent compute-chain graphics-independent \
         graphics-chain mixed-independent mixed-chain; do
  for p in automatic automatic-scoped hazard-only hazard-only-scoped \
           explicit-all explicit-all-scoped; do
    VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
    VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT \
      ./target/release/examples/sync-bench \
        --workload $w --policy $p --passes 16 \
        --warmups 0 --samples 3 --validation 2>&1 |
      grep -E "SYNC-HAZARD|Validation Error" && echo "FAILED: $w/$p"
  done
done
```

Retain the layer output next to the collection. A `SYNC-HAZARD-*` message means
the derived stage or access masks are too narrow for that workload, and a
`Validation Error` means they are inconsistent with the declared stages; either
way the policy must not feed performance data until it is fixed. This check
found a real defect once already: the `extra_sync_*_access` driver workarounds
name transfer accesses that only `ALL_COMMANDS` supports.

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
  --repetitions 5 \
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
  --repetitions 5 `
  --cpu-only
```

macOS Apple Silicon:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend metal \
  --wgpu-adapter-name "Apple M3" \
  --output paper/data/raw/macos-apple-m3-cpu \
  --repetitions 5 \
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
  --repetitions 5
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
  --repetitions 5
```

macOS Apple Silicon:

```sh
python3 paper/run-study-matrix.py \
  --wgpu ../wgpu \
  --backend metal \
  --wgpu-adapter-name "Apple M3" \
  --output paper/data/raw/macos-apple-m3-gpu \
  --repetitions 5
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

## Lock the clocks first

Every collection in this study so far ran with default power management, and
the cost is a control floor -- the disagreement between two configurations that
emit identical commands -- ranging from 0.0% to 48% depending on the cell. A
cell whose floor exceeds the effect you are chasing cannot answer the question.

Short workloads on large idle GPUs are the worst case: the RX 7900 XT never
leaves its idle clock state during a 200-microsecond command buffer, so its
compute cells land at 3.6-6.5% while its graphics cells, which run longer, land
at 0.1-0.8%. Locking clocks is the cheap fix; raising `--rounds` or `--passes`
until the device settles is the other.

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
frequency. Restore with `auto` / `--reset-gpu-clocks` afterwards, and record
which command was used next to the collection.

Then check the floor before believing anything: `build-tables.py` prints a
`ctrl` column per device, and any effect smaller than it is not a result.

## Barrier policies and workloads

On Vulkan the collector runs four Blade policies per workload:

On Vulkan the collector crosses three placements with two scopes, giving six
Blade policies per workload:

| Policy | Placement | Scope |
|---|---|---|
| `automatic` | encoder, every pass | `Global` |
| `automatic-scoped` | encoder, every pass | `PassKind` |
| `hazard-only` | application, at real hazards | `Global` |
| `hazard-only-scoped` | application, at real hazards | `PassKind` |
| `explicit-all` | application, every pass | `Global` |
| `explicit-all-scoped` | application, every pass | `PassKind` |

Reading down a scope column isolates placement; reading across a placement row
isolates scope. `explicit-all` against `automatic` at the same scope is the
instrumentation control and should be indistinguishable. Metal collects
`automatic` only, since `barrier_scope` and `manual_barriers` are Vulkan
concepts.

Workloads are the four shared ones plus `mixed-independent` and `mixed-chain`,
which alternate compute and render passes. The mixed pair is Blade-only, so
wgpu still runs four workloads per repetition and Blade runs six.

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

`paper/profile-hosts.py` attributes host CPU time to the crate or library that
spent it, which is what turns "wgpu costs 2-12x more host time" into a
statement about *what* costs it.

```sh
sudo sysctl kernel.perf_event_paranoid=1     # until reboot
python3 paper/profile-hosts.py --wgpu ../wgpu
```

It refuses to run rather than emit an empty profile if `perf` is missing or
`perf_event_paranoid` is above 2. The workload it profiles is deliberately
mis-shaped for the GPU and well shaped for the host -- 4096 elements, 64x64
targets, one mixing round, 64 passes, no timestamp queries -- so the process
stays in the recording path instead of blocking in `wait_for`. It rebuilds both
binaries with `CARGO_PROFILE_RELEASE_DEBUG=line-tables-only`, which improves
symbol attribution without changing optimisation.

Output is `symbols.csv` (self time per symbol), `buckets.csv` (self time per
group), the raw `perf report` text, and a manifest. The groups separate
`wgpu_core::track` from `wgpu_core::init_tracker`: the first is resource state
tracking, the second is lazy zero-initialisation, and conflating them would
overstate the number the profile exists to measure.

Two caveats to record with the result. Self time is attributed to the symbol
the sample landed in, so aggressive inlining moves tracker work into its
callers and the tracker share is a lower bound. And a flat profile answers
"where is time spent", not "why" -- a call-graph run (`--call-graph dwarf`)
costs more and needs frame pointers to be reliable.

## GPU captures

`paper/capture-streams.py` captures the Vulkan command stream of each
configuration, so the barriers the paper describes from source can be checked
against what reaches the driver.

```sh
python3 paper/capture-streams.py
```

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
masks decoded to names, barrier counts by kind, and any image layout
transition. That is what makes the captures checkable rather than merely
archived.

One gap remains: it captures Blade only. The matched wgpu benchmark has no
`--capture` flag, and adding one that wraps the same warmed iteration would
make the comparison symmetric.

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
/tmp/metal-hazard-bench > paper/data/raw/<collection-id>/metal-hazard.csv
```

It emits a comment header naming the device and OS build, then one row per
(workload, tracking mode, pass count) with median and 5th/95th-percentile
encode, commit, GPU, and wall times. It also reports each buffer's effective
`hazardTrackingMode`, so a run that silently fell back to tracked mode is
detectable from its own output.

The results currently cited by the paper are in
[`../docs/metal-hazard-tracking.md`](../docs/metal-hazard-tracking.md) and were
collected on battery with Low Power Mode enabled. Repeat on AC power with Low
Power Mode disabled before publishing absolute times, and retain the raw
program output, macOS build, hardware report, and Blade revision.

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
python3 paper/run-study-matrix.py --wgpu ../wgpu --repetitions 3
```

Passing `--blade-device-id` or `--wgpu-adapter-name` pins one device and
restores the previous single-collection behaviour. Either way the collector
aborts if Blade and wgpu end up reporting different `device_name` values, which
is the failure the `rubik` collection hit before this check existed.
