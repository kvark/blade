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

## CPU profiles and GPU captures

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

`docs/metal-hazard-tracking.md` references
`tools/metal-hazard-bench.swift`, but that harness is currently absent from the
checkout. Recover it before treating the Apple tables as reproducible. Its
documented build and run commands are:

```sh
xcrun swiftc -O tools/metal-hazard-bench.swift -o /tmp/metal-hazard-bench
/tmp/metal-hazard-bench
```

Run on AC power with Low Power Mode disabled and retain the raw program output,
macOS build, hardware report, and Blade revision.
